; child-snack task with 1 children and 0.77 gluten factor 
; constant factor of 1.9
; random seed: 900192

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child421 - child
    bread270 - bread-portion
    content25 - content-portion
    tray161 tray97 tray245 tray59 - tray
    table484 table416 table327 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray161 kitchen)
     (at tray97 kitchen)
     (at tray245 kitchen)
     (at tray59 kitchen)
     (at_kitchen_bread bread270)
     (at_kitchen_content content25)
     (not_allergic_gluten child421)
     (waiting child421 table327)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child421)
    )
  )
)
