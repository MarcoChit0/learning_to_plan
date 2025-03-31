; child-snack task with 2 children and 0.78 gluten factor 
; constant factor of 1.5
; random seed: 361200

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child312 child42 - child
    bread108 bread491 - bread-portion
    content438 content355 - content-portion
    tray182 tray456 tray421 tray314 - tray
    table243 table212 table179 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray182 kitchen)
     (at tray456 kitchen)
     (at tray421 kitchen)
     (at tray314 kitchen)
     (at_kitchen_bread bread108)
     (at_kitchen_bread bread491)
     (at_kitchen_content content438)
     (at_kitchen_content content355)
     (no_gluten_bread bread491)
     (no_gluten_content content438)
     (allergic_gluten child42)
     (not_allergic_gluten child312)
     (waiting child312 table179)
     (waiting child42 table243)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child312)
     (served child42)
    )
  )
)
