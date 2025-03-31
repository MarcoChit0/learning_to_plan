; child-snack task with 1 children and 0.4 gluten factor 
; constant factor of 1.5
; random seed: 612679

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child404 - child
    bread141 - bread-portion
    content287 - content-portion
    tray279 tray23 - tray
    table312 table149 table393 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray279 kitchen)
     (at tray23 kitchen)
     (at_kitchen_bread bread141)
     (at_kitchen_content content287)
     (not_allergic_gluten child404)
     (waiting child404 table393)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child404)
    )
  )
)
