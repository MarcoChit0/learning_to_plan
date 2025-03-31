; child-snack task with 2 children and 0.27 gluten factor 
; constant factor of 1.7
; random seed: 337762

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child285 child119 - child
    bread259 bread229 - bread-portion
    content298 content101 - content-portion
    tray144 tray286 tray449 - tray
    table352 table46 table113 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray144 kitchen)
     (at tray286 kitchen)
     (at tray449 kitchen)
     (at_kitchen_bread bread259)
     (at_kitchen_bread bread229)
     (at_kitchen_content content298)
     (at_kitchen_content content101)
     (not_allergic_gluten child119)
     (not_allergic_gluten child285)
     (waiting child285 table352)
     (waiting child119 table46)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child285)
     (served child119)
    )
  )
)
