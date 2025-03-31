; child-snack task with 1 children and 0.43 gluten factor 
; constant factor of 1.9
; random seed: 842146

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child56 - child
    bread327 - bread-portion
    content295 - content-portion
    tray5 tray452 tray494 tray77 - tray
    table222 table312 table305 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray5 kitchen)
     (at tray452 kitchen)
     (at tray494 kitchen)
     (at tray77 kitchen)
     (at_kitchen_bread bread327)
     (at_kitchen_content content295)
     (not_allergic_gluten child56)
     (waiting child56 table222)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child56)
    )
  )
)
