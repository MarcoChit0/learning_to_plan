; child-snack task with 2 children and 0.58 gluten factor 
; constant factor of 1.5
; random seed: 631141

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child45 child128 - child
    bread372 bread440 - bread-portion
    content219 content105 - content-portion
    tray425 tray340 tray247 tray281 - tray
    table182 table161 table445 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray425 kitchen)
     (at tray340 kitchen)
     (at tray247 kitchen)
     (at tray281 kitchen)
     (at_kitchen_bread bread372)
     (at_kitchen_bread bread440)
     (at_kitchen_content content219)
     (at_kitchen_content content105)
     (no_gluten_bread bread440)
     (no_gluten_content content219)
     (allergic_gluten child45)
     (not_allergic_gluten child128)
     (waiting child45 table182)
     (waiting child128 table445)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child45)
     (served child128)
    )
  )
)
