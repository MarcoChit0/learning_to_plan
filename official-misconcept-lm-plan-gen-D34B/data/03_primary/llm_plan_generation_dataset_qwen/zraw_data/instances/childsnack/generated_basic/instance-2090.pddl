; child-snack task with 1 children and 0.62 gluten factor 
; constant factor of 1.5
; random seed: 186300

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child108 - child
    bread177 - bread-portion
    content244 - content-portion
    tray431 tray283 tray378 tray262 - tray
    table68 table494 table260 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray431 kitchen)
     (at tray283 kitchen)
     (at tray378 kitchen)
     (at tray262 kitchen)
     (at_kitchen_bread bread177)
     (at_kitchen_content content244)
     (not_allergic_gluten child108)
     (waiting child108 table260)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child108)
    )
  )
)
