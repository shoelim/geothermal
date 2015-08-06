/* Disclaimer: This program is developed by modifying the existing tutorial programs
               in deal.II documentation. A large parts of the codes in this program
               are inherited from the tutorial programs for the purpose of solving
               a problem in an application                 */

/* Copyright (C) 2009-2012 by deal.II authors              */

/* Instruction: To perform numerical experiments, either change the variables in
   parameter.prm or in the namespace SillData @line 78     */

#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/data_out_base.h>
#include <base/quadrature_lib.h>
#include <base/function.h>
#include <base/logstream.h>
#include <base/utilities.h>
#include <base/smartpointer.h>
#include <base/timer.h>
#include <lac/vector.h>
#include <lac/full_matrix.h>
#include <lac/sparse_matrix.h>
#include <lac/solver_cg.h>
#include <lac/compressed_sparsity_pattern.h>
#include <lac/precondition.h>
#include <lac/filtered_matrix.h>
#include <lac/constraint_matrix.h>
#include <grid/tria.h>
#include <grid/grid_generator.h>
#include <grid/grid_tools.h>
#include <grid/tria_accessor.h>
#include <grid/tria_iterator.h>
#include <grid/grid_refinement.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/dofs/dof_tools.h>
#include <dofs/dof_handler.h>
#include <dofs/dof_accessor.h>
#include <dofs/dof_tools.h>
#include <dofs/dof_constraints.h>
#include <dofs/dof_renumbering.h>
#include <fe/fe_q.h>
#include <fe/fe_values.h>
#include <fe/mapping_q.h>
#include <numerics/vectors.h>
#include <numerics/matrices.h>
#include <numerics/data_out.h>
#include <numerics/error_estimator.h>
#include <numerics/solution_transfer.h>
#include <numerics/derivative_approximation.h>

#include <cmath>
#include <fstream>
#include <iostream>
#include <list>
#include <iomanip>
#include <algorithm>

namespace Earth
{
 using namespace dealii;

 namespace EquationConstants
 {                                              /* UNITS             */
    const double sec = (365.25*24*3600);        /* s                 */
    const double k = 8.154421735e19;            /* kg*km/(K*yr^3)    */
    const double rho_cs = 3.05e12;              /* kg/km^3           */
    const double rho_cl = 2.3e12;               /* kg/km^3           */
    const double rho_bs = 3.1e12;               /* kg/km^3           */
    const double rho_bl = 2.83e12;              /* kg/km^3           */
    const double L_b = (4.0e5)*sec*sec/1000000; /* km^2/(K*yr^2)     */
    const double L_c = (3.5e5)*sec*sec/1000000; /* km^2/(K*yr^2)     */
    const double cp_b = 1480*sec*sec/1000000;   /* km^2/yr^2         */
    const double cp_c = 1390*sec*sec/1000000;   /* km^2/yr^2         */
    double kstar = 0;
 }

 namespace SillData
 {
   const double DomainLength = 2.5;             /* km                */
   const double DomainHeight = 1.0;             /* km                */
   const double L1 = 1.0;                       /* km                */
   const double L2 = 1.5;                       /* km                */
   const double W1 = 0.475;                     /* km                */
   const double W2 = 0.525;                     /* km                */
   const double ini_temp = 1558.;               /* K                 */
   const double geotherm = 0.;                 /* K/km              */
   const double back_temp = 873.;               /* K                 */
 }

namespace Parameters
 {
   enum TestCase
   {
      TEST_ZERO,
      TEST_ONE
   };

   class DataInput
   {
     public:
        DataInput ();
        ~DataInput ();
        void read_data (const char *filename);
        TestCase test;
        double time,
        final_time,
        time_step,
        TC_liquid,
        TC_solid,
        TM_liquid,
        TM_interm,
        TM_solid,
        dm_meltfrac;
        unsigned int deg;
        unsigned int freq,
        mode,
        ini_refinement_level,
        n_pre_refinement_steps,
        level;
     protected:
        ParameterHandler prm;
   };

   DataInput::DataInput()
   {
        prm.declare_entry ("TestCase", "TEST_ONE",
                           Patterns::Selection ("TEST_ZERO|TEST_ONE"),
                           "Used to select the test case that we are going "
                           "to use. ");

        prm.enter_subsection ("Physical data");
        {
           prm.declare_entry ("time", "0.", Patterns::Double (0.),
                              "Time of simulation. ");
           prm.declare_entry ("final_time", "0.", Patterns::Double (0.),
                              "Final time of simulation. ");
           prm.declare_entry ("time_step", "0.", Patterns::Double (0.),
                              "Time step of simulation. ");
           prm.declare_entry ("TC_solid", "0.", Patterns::Double (0.),
                              "Solid temp for crust. ");
           prm.declare_entry ("TC_liquid", "0.", Patterns::Double (0.),
                              "Liquid temp for crust. ");
           prm.declare_entry ("TM_solid", "0.", Patterns::Double (0.),
                              "Solid temp for sill. ");
           prm.declare_entry ("TM_interm", "0.", Patterns::Double (0.),
                              "Intermediate temp for sill. ");
           prm.declare_entry ("TM_liquid", "0.", Patterns::Double (0.),
                              "Liquid temp for sill. ");

        }
        prm.leave_subsection();

        prm.enter_subsection ("Space discretization");
        {
           prm.declare_entry ("deg", "1", Patterns::Integer (1,3),
                              "Polynomial degree for the temperature space");
           prm.declare_entry ("freq", "0", Patterns::Integer (0,10000),
                              "Frequency of pre-time adaptive refinement");
           prm.declare_entry ("ini_refinement_level", "2",
                              Patterns::Integer (2,1000),
                              "Initial number of cells = this*this");
           prm.declare_entry ("n_pre_refinement_steps", "0",
                              Patterns::Integer (0,10),
                              "Number of pre adaptive refinement steps");
           prm.declare_entry ("mode", "1", Patterns::Integer (1, 5),
                              "Choice of refinement strategy. ");
           prm.declare_entry ("level", "0", Patterns::Integer (0, 10),
                              "Number of global refinements for mode 1. ");
        }
        prm.leave_subsection();

   }

   DataInput::~DataInput()
   {}

   void DataInput::read_data (const char *filename)
   {
        std::ifstream file (filename);
        AssertThrow (file, ExcFileNotOpen (filename));

        prm.read_input (file);

        if (prm.get ("TestCase") == std::string ("TEST_ZERO")){
           test = TEST_ZERO;
        }
        else{
           test = TEST_ONE;
        }

        prm.enter_subsection ("Physical data");
        {
           time = prm.get_double ("time");
           final_time = prm.get_double ("final_time");
           time_step = prm.get_double ("time_step");
           TC_solid = prm.get_double ("TC_solid");
           TC_liquid = prm.get_double ("TC_liquid");
           TM_solid = prm.get_double ("TM_solid");
           TM_interm = prm.get_double ("TM_interm");
           TM_liquid = prm.get_double ("TM_liquid");
        }
        prm.leave_subsection();

        prm.enter_subsection ("Space discretization");
        {
           deg = prm.get_integer ("deg");
           freq = prm.get_integer ("freq");
           ini_refinement_level = prm.get_integer ("ini_refinement_level");
           n_pre_refinement_steps = prm.get_integer ("n_pre_refinement_steps");
           mode = prm.get_integer ("mode");
           level = prm.get_integer ("level");
        }
        prm.leave_subsection();

   }
 }


 /* class denoting the initial conditions for case with zero geotherm */
 class InitialValues_ZeroGeotherm : public Function<2>
 {
   public:
        InitialValues_ZeroGeotherm (const unsigned int n_components = 2,
                const double time = 0.)
                :
                Function<2>(n_components, time)
        {}
        virtual double value (const Point<2> &p,
                const unsigned int component = 0) const;
        virtual void vector_value (const Point<2> &p,
                Vector<double> &value) const;
 };

 double InitialValues_ZeroGeotherm::value (const Point<2> &p,
        const unsigned int /*component*/) const
 {
        const double x = p[0];
        const double y = p[1];

        if(x>=SillData::L1 && x<=SillData::L2 && y>= SillData::W1
                && y<=SillData::W2){
                return SillData::ini_temp;
        }
        else{
                return SillData::back_temp;
        }
 }

 void InitialValues_ZeroGeotherm::vector_value (const Point<2> &p,
        Vector<double> &value) const
 {
        for (unsigned int c=0; c<this->n_components; ++c){
                value(c) = InitialValues_ZeroGeotherm::value(p,c);
        }
 }

 /* class denoting the initial conditions for case with nonzero geotherm */
 class InitialValues_Geotherm : public Function<2>
 {
   public:
        InitialValues_Geotherm (const unsigned int n_components = 2,
                const double time = 0.)
                :
                Function<2>(n_components, time)
        {}
        virtual double value (const Point<2> &p,
                const unsigned int component = 0) const;
        virtual void vector_value (const Point<2> &p,
                Vector<double> &value) const;
 };

 double InitialValues_Geotherm::value (const Point<2> &p,
        const unsigned int /*component*/) const
 {
        const double x = p[0];
        const double y = p[1];

        if(x>=SillData::L1 && x<=SillData::L2 && y>= SillData::W1
                && y<=SillData::W2){
                return SillData::ini_temp;
        }
        else{
                return (SillData::back_temp-SillData::geotherm*SillData::DomainHeight/2) +
                        SillData::geotherm*y;
        }
 }

 void InitialValues_Geotherm::vector_value (const Point<2> &p,
        Vector<double> &value) const
 {
        for (unsigned int c=0; c<this->n_components; ++c){
                value(c) = InitialValues_Geotherm::value(p,c);
        }
 }


 /* class denoting the boundary conditions for case with zero geotherm */
 class BoundaryData_ZeroGeotherm : public Function<2>
 {
    public:
        BoundaryData_ZeroGeotherm (const double time = 0.)  : Function<2>(1, time) {}
        virtual double value (const Point<2>   &p,
                              const unsigned int  component = 0) const;
        virtual void value_list (const std::vector<Point<2> > &points,
                                 std::vector<double>          &values,
                                 const unsigned int component = 0) const;
 };

 double BoundaryData_ZeroGeotherm::value (const Point<2>   & /*p*/,
        const unsigned int /*component*/ ) const
 {
        return SillData::back_temp;
 }

 void BoundaryData_ZeroGeotherm::value_list(const std::vector<Point<2> > &points,
                                std::vector<double>          &values,
                                const unsigned int component) const
 {
         Assert (values.size() == points.size(),
                ExcDimensionMismatch (values.size(), points.size()));
         Assert (component == 0,
                ExcIndexRange (component, 0, 1));

        const unsigned int n_points = points.size();

        for (unsigned int i=0; i<n_points; ++i)
        {
                values[i] = BoundaryData_ZeroGeotherm::value(points[i]);
        }
 }

 /* class denoting the boundary conditions for case with nonzero geotherm */
 class BoundaryData_Geotherm : public Function<2>
 {
    public:
        BoundaryData_Geotherm (const double time = 0.)  : Function<2>(1, time) {}
        virtual double value (const Point<2>   &p,
                              const unsigned int  component = 0) const;
        virtual void value_list (const std::vector<Point<2> > &points,
                                 std::vector<double>          &values,
                                 const unsigned int component = 0) const;
 };

 double BoundaryData_Geotherm::value (const Point<2>   &p,
        const unsigned int /*component*/ ) const
 {
        const double x = p[0];
        const double y = p[1];

        if(x>=SillData::L1 && x<=SillData::L2 && y>= SillData::W1
                && y<=SillData::W2){
                return SillData::ini_temp;
        }
        else{
                return (SillData::back_temp-SillData::geotherm*SillData::DomainHeight/2) +
                        SillData::geotherm*y;
        }
 }

 void BoundaryData_Geotherm::value_list(const std::vector<Point<2> > &points,
                                std::vector<double>          &values,
                                const unsigned int component) const
 {
         Assert (values.size() == points.size(),
                ExcDimensionMismatch (values.size(), points.size()));
         Assert (component == 0,
                ExcIndexRange (component, 0, 1));

        const unsigned int n_points = points.size();

        for (unsigned int i=0; i<n_points; ++i)
        {
                values[i] = BoundaryData_Geotherm::value(points[i]);
        }
 }

 /* class denoting the right hand side */
 class RightHandSide : public Function<2>
 {
   public:
        RightHandSide (double time = 0.)  : Function<2>(1, time) {}
        virtual double value (const Point<2>   &p,
                              const unsigned int  component = 0) const;
        virtual void value_list (const std::vector<Point<2> > &points,
                                 std::vector<double>          &values,
                                 const unsigned int component = 0) const;
 };

 double RightHandSide::value (const Point<2>   &/*p*/,
                              const unsigned int /*component*/ ) const
 {
        return 0;
 }

 void RightHandSide::value_list(const std::vector<Point<2> > &points,
                               std::vector<double>          &values,
                               const unsigned int component) const
 {
        Assert (values.size() == points.size(),
        ExcDimensionMismatch (values.size(), points.size()));
        Assert (component == 0,
        ExcIndexRange (component, 0, 1));

        const unsigned int n_points = points.size();

        for (unsigned int i=0; i<n_points; ++i){
                values[i] = RightHandSide::value(points[i]);
        }
 }

 /* the main class */
 class Model
 {
   public:
     Model (const Parameters::DataInput &data);
     void run ();

   protected:
     Parameters::TestCase test;
     unsigned int deg;
     unsigned int freq, ini_refinement_level;
     unsigned int n_pre_refinement_steps, mode, level;
     double time, time_step;
     const double final_time;
     const double TC_0, TC_1, TM_0, TM_1, TM_i;

   private:
     void make_initial_grid ();
     void setup_dof ();
     void refine_by_Kelly ();
     void refine_by_derivative ();
     void refine_grid (const unsigned int max_grid_level);
     void assemble_term ();
     void compute_oldterm ();
     void compute_newterm ();
     void compute_matrix ();
     void assemble_system ();
     void solve ();
     void output_results (const unsigned int timestep_number) const;
     double evaluate_soln (double x, double y) const ;
     double compute_kstar (double temp, unsigned char id);

     Triangulation<2>            triangulation;
     FE_Q<2>                     fe;
     DoFHandler<2>               dof_handler;

     SparsityPattern             sparsity_pattern;
     SparseMatrix<double>        system_matrix;

     ConstraintMatrix            constraints;

     PreconditionSSOR<>          preconditioner;

     const MappingQ<2>           mapping;

     bool                        rebuild_matrix;

     Vector<double>              solution, old_solution, melt_fraction;
     Vector<double>              system_rhs;

     std::vector<double>         error;
     double                      theta;

     RightHandSide               rhs;
     BoundaryData_ZeroGeotherm   boundarydata1;
     BoundaryData_Geotherm       boundarydata2;
     InitialValues_ZeroGeotherm  initialvalues1;
     InitialValues_Geotherm      initialvalues2;
     TimerOutput                 computing_timer;
     std::ofstream               summary;
 };

 Model::Model (const Parameters::DataInput &data)
    :
    test (data.test),
    freq (data.freq),
    ini_refinement_level (data.ini_refinement_level),
    n_pre_refinement_steps (data.n_pre_refinement_steps),
    mode (data.mode),
    level (data.level),
    time (data.time),
    time_step (data.time_step),
    final_time (data.final_time),
    TC_0 (data.TC_solid),
    TC_1 (data.TC_liquid),
    TM_0 (data.TM_solid),
    TM_1 (data.TM_liquid),
    TM_i (data.TM_interm),
    fe  (data.deg),
    dof_handler (triangulation),
    mapping (4),
    rebuild_matrix (false),
    theta (1.0), /* 0 <= theta <= 1 
                   theta = 0 corresp to explicit Euler scheme - 1st order acc
                   theta = 1 corresp to implicit Euler scheme - 1st order acc
                   theta = 0.5 corresp to Crank-Nicolson scheme - 2nd order acc
                 */
    rhs (time),
    boundarydata1 (time),
    boundarydata2 (time),
    computing_timer (summary, TimerOutput::summary, TimerOutput::wall_times)
 {}

 /* compute k* at a particular time step using previous solution */
 double Model::compute_kstar(double temp, unsigned char id)
 {
      const double k = EquationConstants::k;
      const double cp_c = EquationConstants::cp_c;
      const double cp_b = EquationConstants::cp_b;
      const double L_c = EquationConstants::L_c;
      const double L_b = EquationConstants::L_b;
      const double rho_cs = EquationConstants::rho_cs;
      const double rho_cl = EquationConstants::rho_cl;
      const double rho_bs = EquationConstants::rho_bs;
      const double rho_bl = EquationConstants::rho_bl;
      double kstar = 10000000;
      double denom = 1.;

      if(test == Parameters::TEST_ZERO){
                kstar = k/(rho_cs*cp_c);
      }
      else if(id == 'c'){
         if(temp >= TC_0 && temp <= TC_1){
                double dc_mf = 1/(TC_1 - TC_0);
                double meltf = dc_mf*temp-1.230492197;
                double rho = meltf*(rho_cl - rho_cs) + rho_cs;
                kstar = (k/(rho*cp_c + rho*L_c*dc_mf))/denom;
         }
         else if(temp > TC_1){
                kstar = (k/(rho_cl*cp_c + rho_cl*L_c*1))/denom;
         }
         else{
                kstar = (k/(rho_cs*cp_c + rho_cs*L_c*0))/denom;
         }
      }
      else if(id == 'm'){
         if(temp >= TM_0 && temp <= TM_i){
                double dm_mf = 1.4e-3;
                double meltf = dm_mf*temp - 1.3986;
                double rho = meltf*(rho_bl - rho_bs) + rho_bs;
                kstar = (k/(rho*cp_b + rho*L_b*dm_mf))/denom;
         }
         else if(temp >= TM_i && temp <= TM_1){
                double dm_mf = 3.307482993e-3;
                double meltf = dm_mf*temp - 4.004221768;
                double rho = meltf*(rho_bl - rho_bs) + rho_bs;
                kstar = (k/(rho*cp_b + rho*L_b*dm_mf))/denom;
         }
         else if(temp > TM_1){
                kstar = (k/(rho_bl*cp_b + rho_bl*L_b*1))/denom;
         }
         else{
                kstar = (k/(rho_bs*cp_b + rho_bs*L_b*0))/denom;
         }
      }
      else{
                std::cout << " Error in allocating subdomains! " << std::endl;
      }

      return kstar;
  }

  /* adaptive refinement with KellyErrorEstimator */
  void Model::refine_by_Kelly ()
  {
      std::cout << "======= Refined adaptively using KellyErrorEstimator ======="
                  <<  std::endl;
      Vector<float> estimated_error_per_cell (triangulation.n_active_cells());

      KellyErrorEstimator<2>::estimate (mapping, dof_handler,
                                        QGauss<1>(3),
                                        FunctionMap<2>::type(),
                                        solution,
                                        estimated_error_per_cell);

      if(time == 0){
          GridRefinement::refine_and_coarsen_fixed_number (triangulation,
                                                           estimated_error_per_cell,
                                                           0.3, 0.1);
      }
      else if (time < 1000*time_step){
          GridRefinement::refine_and_coarsen_fixed_number (triangulation,
                                                           estimated_error_per_cell,
                                                           0.005, 0.0025);
      }
      else{
         GridRefinement::refine_and_coarsen_fixed_number (triangulation,
                                                          estimated_error_per_cell,
                                                          0.003, 0.0025);
      }
  }


  /* adaptive refinement with DerivativeApproximation */
  void Model::refine_by_derivative ()
  {
      std::cout << "======= Refined adaptively using DerivativeApproximation ======="
                  <<  std::endl;

      Vector<float> gradient_indicator (triangulation.n_active_cells());

      DerivativeApproximation::approximate_second_derivative (mapping,
                                                              dof_handler,
                                                              solution,
                                                              gradient_indicator);

      DoFHandler<2>::active_cell_iterator
      cell = dof_handler.begin_active(),
      endc = dof_handler.end();
      for (unsigned int cell_no=0; cell!=endc; ++cell, ++cell_no)
          gradient_indicator(cell_no)*=std::pow(cell->diameter(), 3.);

          if(time == 0){
                GridRefinement::refine_and_coarsen_fixed_number (triangulation,
                                                                 gradient_indicator,
                                                                 0.3, 0.1);
          }
          else if (time < 1000*time_step){
                GridRefinement::refine_and_coarsen_fixed_number (triangulation,
                                                                 gradient_indicator,
                                                                 0.005, 0.0025);
          }
          else{
                GridRefinement::refine_and_coarsen_fixed_number (triangulation,
                                                                 gradient_indicator,
                                                                 0.002, 0.0025);
         }
  }

  /* main function for adaptive mesh refinement*/
  void Model::refine_grid (const unsigned int max_grid_level)
  {
     computing_timer.enter_section ("Remeshing");

     if(mode == 1){
        std::cout << "======= Refined globally " << level << " times  ======="
                   <<  std::endl;
     }
     else if(mode == 2){
        refine_by_Kelly();
     }
     else{
        refine_by_derivative();
     }

     if(triangulation.n_levels() > max_grid_level)
        for(Triangulation<2>::active_cell_iterator
                cell = triangulation.begin_active(max_grid_level);
                cell != triangulation.end(); ++cell){
                        cell->clear_refine_flag();
        }

        SolutionTransfer<2> soltrans (dof_handler);


        /* Transfer old soln to new mesh */
        triangulation.prepare_coarsening_and_refinement();

        soltrans.prepare_for_coarsening_and_refinement (solution);
        triangulation.execute_coarsening_and_refinement ();

        dof_handler.distribute_dofs(fe);

        constraints.clear();
        DoFTools::make_hanging_node_constraints (dof_handler, constraints);
        constraints.close();

        system_matrix.clear();

        CompressedSparsityPattern c_sparsity (dof_handler.n_dofs());
        DoFTools::make_sparsity_pattern (dof_handler, c_sparsity);
        constraints.condense (c_sparsity);
        sparsity_pattern.copy_from(c_sparsity);


        Vector<double> interpolated_soln(dof_handler.n_dofs());
        soltrans.interpolate(solution, interpolated_soln);
        solution = interpolated_soln;

        system_matrix.reinit(sparsity_pattern);
        old_solution.reinit(dof_handler.n_dofs());
        system_rhs.reinit(dof_handler.n_dofs());

        rebuild_matrix = true;

        computing_timer.exit_section();
  }
  

  /* set up all data structures needed for computation */
  void Model::setup_dof ()
  {
        computing_timer.enter_section("Setup dof systems");

        dof_handler.distribute_dofs (fe);

        constraints.clear ();
        DoFTools::make_hanging_node_constraints (dof_handler, constraints);
        constraints.close ();

        system_matrix.clear();

        CompressedSparsityPattern c_sparsity (dof_handler.n_dofs());
        DoFTools::make_sparsity_pattern (dof_handler, c_sparsity, constraints,
                                         false);
        sparsity_pattern.copy_from(c_sparsity);

        system_matrix.reinit(sparsity_pattern);

        solution.reinit (dof_handler.n_dofs());
        old_solution.reinit(dof_handler.n_dofs());
        system_rhs.reinit (dof_handler.n_dofs());

        /* Optional: display sparsity of global matrix */
        // std::ofstream out("sparsity_pattern.m2");
        // sparsity_pattern.print_gnuplot(out);
        computing_timer.exit_section();
  }

  /* compute the main matrix and right hand side */
  void Model::assemble_term ()
  {
        computing_timer.enter_section ("   Assembling system");
        system_rhs = 0;
        compute_oldterm ();
        rhs.advance_time (time_step);
        compute_newterm ();
        computing_timer.exit_section();
  }


  void Model::compute_oldterm ()
  {
        const QGauss<2> quadrature_formula (3);
        FEValues<2>     fe_values (mapping, fe, quadrature_formula,
                                   update_values | update_gradients |
                                   update_JxW_values | update_q_points);

        const unsigned int dofs_per_cell = dof_handler.get_fe().dofs_per_cell;
        const unsigned int n_q_points    = quadrature_formula.size();
        double coeff = 0.;
        double cont = 1.;

        Vector<double> local_term (dofs_per_cell);
        std::vector<unsigned int> local_dof_indices (dofs_per_cell);
        std::vector<double> old_data_values (n_q_points);
        std::vector<Tensor<1,2> > old_data_grads (n_q_points);

        std::vector<double> rhs_values_old (n_q_points);

        DoFHandler<2>::active_cell_iterator
        cell = dof_handler.begin_active(),
        endc = dof_handler.end();

        for(; cell!=endc; ++cell){
                local_term = 0;
                fe_values.reinit (cell);
                fe_values.get_function_values (old_solution, old_data_values);
                fe_values.get_function_grads (old_solution, old_data_grads);
                rhs.value_list(fe_values.get_quadrature_points(), rhs_values_old);

        for(unsigned int q_point=0; q_point<n_q_points; ++q_point){
                coeff = compute_kstar(old_data_values[q_point], cell->material_id());

           for(unsigned int i=0; i<dofs_per_cell; ++i){
                local_term(i) += (1/(time_step)*old_data_values[q_point]*
                                 fe_values.shape_value(i,q_point) + (1-theta)*
                                 rhs_values_old[q_point]*
                                 fe_values.shape_value(i, q_point) - coeff*cont*
                                 (1-theta)*old_data_grads[q_point]*
                                 fe_values.shape_grad(i,q_point))*
                                 fe_values.JxW (q_point);
           }
        }

        cell->get_dof_indices (local_dof_indices);

           for (unsigned int i=0; i<dofs_per_cell; ++i){
                system_rhs(local_dof_indices[i]) += local_term(i);
           }
        }
        constraints.condense (system_rhs);
  }

  void Model::compute_newterm ()
  {
        const QGauss<2> quadrature_formula (3);
        FEValues<2>     fe_values (mapping, fe, quadrature_formula,
                                   update_values | update_gradients |
                                   update_JxW_values | update_q_points);

        const unsigned int dofs_per_cell = fe.dofs_per_cell;
        const unsigned int n_q_points    = quadrature_formula.size();

        Vector<double> local_term (dofs_per_cell);
        std::vector<unsigned int> local_dof_indices (dofs_per_cell);

        std::vector<double> rhs_values_new (n_q_points);

        DoFHandler<2>::active_cell_iterator
        cell = dof_handler.begin_active(),
        endc = dof_handler.end();

        for(; cell!=endc; ++cell){
                local_term = 0;
                fe_values.reinit (cell);
                rhs.value_list(fe_values.get_quadrature_points(), rhs_values_new);

           for (unsigned int q_point=0; q_point<n_q_points; ++q_point){
                for (unsigned int i=0; i<dofs_per_cell; ++i){
                        local_term(i) += theta*rhs_values_new[q_point]*
                                         fe_values.shape_value(i,q_point)*
                                         fe_values.JxW (q_point);
                }
           }

           cell->get_dof_indices (local_dof_indices);

           for (unsigned int i=0; i<dofs_per_cell; ++i){
                system_rhs(local_dof_indices[i]) += local_term(i);
           }

        }
        constraints.condense(system_rhs);
  }

  /* compute initial matrix */
  void Model::compute_matrix ()
  {
        computing_timer.enter_section ("   Rebuilding matrix");

        system_matrix = 0;

        QGauss<2>   quadrature_formula (3);
        FEValues<2> fe_values (mapping, fe, quadrature_formula,
                               update_values | update_gradients | update_JxW_values | 
                               update_q_points);

        const unsigned int dofs_per_cell = fe.dofs_per_cell;
        const unsigned int n_q_points    = quadrature_formula.size();
        double coeff = 0;
        double cont = 1.;

        FullMatrix<double> local_matrix (dofs_per_cell, dofs_per_cell);
        std::vector<unsigned int> local_dof_indices (dofs_per_cell);
        std::vector<double> old_data_values (n_q_points);

        DoFHandler<2>::active_cell_iterator
        cell = dof_handler.begin_active(),
        endc = dof_handler.end();

        for (; cell!=endc; ++cell){
                local_matrix = 0;
                fe_values.reinit (cell);
                fe_values.get_function_values (old_solution, old_data_values);

          for (unsigned int q_point=0; q_point<n_q_points; ++q_point){
                coeff = compute_kstar(old_data_values[q_point], cell->material_id());
            for (unsigned int i=0; i<dofs_per_cell; ++i){
                for (unsigned int j=0; j<dofs_per_cell; ++j){
                        local_matrix(i,j) += (1/(time_step)*
                                             fe_values.shape_value(i,q_point)*
                                             fe_values.shape_value(j,q_point) +
                                             theta*coeff*cont*
                                             fe_values.shape_grad(i, q_point)*
                                             fe_values.shape_grad(j, q_point))*
                                             fe_values.JxW (q_point);
                }
            }
         }

         cell->get_dof_indices (local_dof_indices);

         for (unsigned int i=0; i<dofs_per_cell; ++i){
             for (unsigned int j=0; j<dofs_per_cell; ++j){
                 system_matrix.add(local_dof_indices[i], local_dof_indices[j],
                                   local_matrix(i,j));
             }
         }
       }

       constraints.condense (system_matrix);
       computing_timer.exit_section();
  }


  /* rebuilding matrix if necessary */
  void Model::assemble_system ()
  {
        if(rebuild_matrix == true){
                compute_matrix();
        }
        assemble_term ();
  }

  /* solve the resulting system with a cg solver */
  void Model::solve ()
  {
        computing_timer.enter_section ("   Solve system");

        SolverControl solver_control (1000, 1e-12, false, false);
        SolverCG<> cg (solver_control);
        preconditioner.initialize(system_matrix, 1.2);

        FilteredMatrix<Vector<double> > f_matrix(system_matrix);
        std::map<unsigned int,double> boundary_values;

        if(SillData::geotherm == 0){
        VectorTools::interpolate_boundary_values (dof_handler,
                                                  0,
                                                  boundarydata1,
                                                  boundary_values);
        }
        else{
                VectorTools::interpolate_boundary_values (dof_handler,
                                                          0,
                                                          boundarydata2,
                                                          boundary_values);
        }

        f_matrix.add_constraints(boundary_values);
        f_matrix.apply_constraints(system_rhs, true);
        cg.solve (f_matrix, solution, system_rhs, preconditioner);
        constraints.distribute (solution);
        std::cout << solver_control.last_step()
                  << " CG iterations."
                  << std::endl;
        computing_timer.exit_section();
  }


  /* output results with gnuplot */
  void Model::output_results (const unsigned int timestep_number) const
  {
        DataOut<2> data_out_one;
        data_out_one.attach_dof_handler (dof_handler);
        data_out_one.add_data_vector (solution, "u");
        data_out_one.build_patches ();
        std::string name = "not_the_solution.";

        switch (test){
                case Parameters::TEST_ZERO:
                name = "fsolution.TEST0.";
                break;

                case Parameters::TEST_ONE:
                name = "fsolution.TEST1.";
                break;

                default:
                Assert (false, ExcNotImplemented());
        };

        std::string tname = "not";

        tname = "0.";

        const std::string filename1 = name + tname  +
                                      Utilities::int_to_string (timestep_number, 5) +
                                      ".gnuplot";

        std::ofstream output_one (filename1.c_str());
        data_out_one.write_gnuplot (output_one);
  }


  /* evaluate numerical solution at a particular point */
  double Model::evaluate_soln (double x, double y) const
  {
        return VectorTools::point_value (mapping, dof_handler, solution,
                                         Point<2>(x, y));
  }

  /* create initial uniform grid */
  void Model::make_initial_grid(){
        const Point<2> botleft = (Point<2> (0., 0.));
        const Point<2> upright = (Point<2> (SillData::DomainLength,
                                  SillData::DomainHeight));

        std::vector<unsigned int> n_div;
        n_div.push_back(ini_refinement_level);
        n_div.push_back(ini_refinement_level);

        GridGenerator::subdivided_hyper_rectangle(triangulation, n_div,
                                                  botleft, upright);
  }


 /* run the simulation in time */
 void Model::run ()
 {
    std::cout << "============================ Running TEST_" << test
              << std::endl;
    std::cout << "============================ Geotherm = " << SillData::geotherm
              << std::endl;
    std::cout << "Time step #0" << std::endl;

    /* open file to read temperature solutions */
    std::ofstream file_one;

    std::string name, tname, tmode;
    if(test == Parameters::TEST_ZERO){
        name = "T0_";
    }
    else{
        name = "T1_";
    }

    tname = "0_";

    std::string filename1 = name + tname  + "soln.txt";

    file_one.open(filename1.c_str());

    make_initial_grid();

    if(mode == 1){
        triangulation.refine_global(level);
    }

    setup_dof();

    unsigned int pre_refinement_step = 0;

    start_time_iteration:

    time = 0.;
    unsigned int timestep_number = 1;

    /* assign material id to each cell */
    if(test != Parameters::TEST_ZERO){
        for (Triangulation<2>::active_cell_iterator
             cell = triangulation.begin_active();
             cell!=triangulation.end();
             ++cell){
                if(cell->center()[0] <= SillData::L2 && cell->center()[0]
                        >= SillData::L1 && cell->center()[1] <= SillData::W2 &&
                        cell->center()[1] >= SillData::W1){
                        cell->recursively_set_material_id('m');
                 }
                 else{
                        cell->recursively_set_material_id('c');
                }
             }
    }

    compute_matrix();

    /* project initial conditions onto initial mesh */
    if(SillData::geotherm == 0){
        VectorTools::project (mapping, dof_handler,
                constraints,
                QGauss<2>(4),
                InitialValues_ZeroGeotherm (1, time),
                solution);
    }
    else{
         VectorTools::project (mapping, dof_handler,
                constraints,
                QGauss<2>(4),
                InitialValues_Geotherm (1, time),
                solution);
   }



   if(mode != 1 && timestep_number == 1 &&
        pre_refinement_step < n_pre_refinement_steps){
                refine_grid(n_pre_refinement_steps);
                ++pre_refinement_step;
                goto start_time_iteration;
   }

   output_results (0);
   /* loop over time to compute temperature solutions */
   for(time+=time_step; time<=final_time; time+=time_step, ++timestep_number){

        std::cout << "Time step dt = " << time_step << std::endl;

        std::cout << "Number of active cells: "
        << triangulation.n_active_cells()
        << std::endl
        << "Total number of cells: "
        << triangulation.n_cells()
        << std::endl
        << "Total dofs: " << dof_handler.n_dofs()
        << std::endl;

        /* adaptive mesh refinement every "freq" time steps */
        if(mode != 1 && (timestep_number % freq == 0)){
                refine_grid(n_pre_refinement_steps);
        }

        if(SillData::geotherm == 0){
                boundarydata1.advance_time(time_step);
        }
        else{
                boundarydata2.advance_time(time_step);
        }

        if(timestep_number == 1){
                file_one << timestep_number-1 << " "
                         << dof_handler.n_dofs()    
                         << " " << evaluate_soln(1.25,0.525)   
                         << " " << evaluate_soln(1.25,0.530)
                         << " " << evaluate_soln(1.5,0.525)
                         << " " << evaluate_soln(1.5,0.530)
                         << std::endl;
        }

        old_solution = solution;

        std::cout << std::endl
                  << "Time step #" << timestep_number << "; "
                  << "advancing to t = " << time << "."
                  << std::endl;

        assemble_system ();

        solve ();

        output_results (timestep_number);

        if(timestep_number >= 1){
                file_one << timestep_number
                         << " " << dof_handler.n_dofs()
                         << " " << evaluate_soln(1.25,0.525)
                         << " " << evaluate_soln(1.25,0.530)
                         << " " << evaluate_soln(1.5,0.525)
                         << " " << evaluate_soln(1.5,0.530)
                         << std::endl;

        }

        /* stop the program when the entire crust domain solidifies */
        if(evaluate_soln((SillData::L1 + SillData::L2)/2,
                (SillData::W1 + SillData::W2)/2 ) < TM_0){
                        summary.open("Timer.txt");
                        computing_timer.print_summary ();
                        summary.close();
                        file_one << "Solidification time : " << time << std::endl;
                        file_one.close();
                        return;
        }
    }

  }
/* end of Earth namespace */
}


int main ()
{
  using namespace dealii;
  using namespace Earth;

  try
  {
    /* read inputs from parameter.prm */
    Parameters::DataInput data;
    data.read_data ("parameter.prm");

    deallog.depth_console (0);

    Model model(data);
    model.run ();
  }
  catch (std::exception &exc)
   {
    std::cerr << std::endl << std::endl
      << "----------------------------------------------------"
      << std::endl;
    std::cerr << "Exception on processing: " << std::endl
      << exc.what() << std::endl
      << "Aborting!" << std::endl
      << "----------------------------------------------------"
      << std::endl;

    return 1;
  }
  catch (...)
  {
    std::cerr << std::endl << std::endl
      << "----------------------------------------------------"
      << std::endl;
    std::cerr << "Unknown exception!" << std::endl
      << "Aborting!" << std::endl
      << "----------------------------------------------------"
      << std::endl;
    return 1;
  }

  return 0;
}
