; child-snack task with 1 children and 0.49 gluten factor 
; constant factor of 1.7
; random seed: 963578

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child45 - child
    bread492 - bread-portion
    content246 - content-portion
    tray385 tray133 - tray
    table47 table230 table292 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray385 kitchen)
     (at tray133 kitchen)
     (at_kitchen_bread bread492)
     (at_kitchen_content content246)
     (not_allergic_gluten child45)
     (waiting child45 table292)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child45)
    )
  )
)
