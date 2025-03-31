; child-snack task with 2 children and 0.8 gluten factor 
; constant factor of 1.7
; random seed: 725949

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child304 child214 - child
    bread136 bread315 - bread-portion
    content298 content102 - content-portion
    tray285 tray112 - tray
    table496 table212 table378 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray285 kitchen)
     (at tray112 kitchen)
     (at_kitchen_bread bread136)
     (at_kitchen_bread bread315)
     (at_kitchen_content content298)
     (at_kitchen_content content102)
     (no_gluten_bread bread136)
     (no_gluten_content content298)
     (allergic_gluten child304)
     (not_allergic_gluten child214)
     (waiting child304 table496)
     (waiting child214 table212)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child304)
     (served child214)
    )
  )
)
