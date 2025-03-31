; child-snack task with 2 children and 0.32 gluten factor 
; constant factor of 1.9
; random seed: 231037

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child382 child177 - child
    bread312 bread146 - bread-portion
    content389 content312 - content-portion
    tray22 tray94 - tray
    table133 table321 table209 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray22 kitchen)
     (at tray94 kitchen)
     (at_kitchen_bread bread312)
     (at_kitchen_bread bread146)
     (at_kitchen_content content389)
     (at_kitchen_content content312)
     (not_allergic_gluten child382)
     (not_allergic_gluten child177)
     (waiting child382 table133)
     (waiting child177 table133)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child382)
     (served child177)
    )
  )
)
