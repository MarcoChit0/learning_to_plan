; child-snack task with 2 children and 0.38 gluten factor 
; constant factor of 1.9
; random seed: 808471

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child32 child197 - child
    bread138 bread296 - bread-portion
    content289 content199 - content-portion
    tray198 - tray
    table206 table81 table5 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray198 kitchen)
     (at_kitchen_bread bread138)
     (at_kitchen_bread bread296)
     (at_kitchen_content content289)
     (at_kitchen_content content199)
     (not_allergic_gluten child32)
     (not_allergic_gluten child197)
     (waiting child32 table206)
     (waiting child197 table5)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child32)
     (served child197)
    )
  )
)
