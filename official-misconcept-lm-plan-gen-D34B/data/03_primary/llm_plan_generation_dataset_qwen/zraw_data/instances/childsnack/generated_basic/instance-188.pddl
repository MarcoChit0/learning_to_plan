; child-snack task with 2 children and 0.31 gluten factor 
; constant factor of 1.7
; random seed: 392984

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child280 child453 - child
    bread9 bread26 - bread-portion
    content204 content494 - content-portion
    tray236 tray457 tray421 - tray
    table232 table461 table46 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray236 kitchen)
     (at tray457 kitchen)
     (at tray421 kitchen)
     (at_kitchen_bread bread9)
     (at_kitchen_bread bread26)
     (at_kitchen_content content204)
     (at_kitchen_content content494)
     (not_allergic_gluten child453)
     (not_allergic_gluten child280)
     (waiting child280 table46)
     (waiting child453 table46)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child280)
     (served child453)
    )
  )
)
