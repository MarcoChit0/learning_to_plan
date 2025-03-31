; child-snack task with 1 children and 0.49 gluten factor 
; constant factor of 1.7
; random seed: 132156

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child265 - child
    bread141 - bread-portion
    content495 - content-portion
    tray204 tray152 - tray
    table361 table156 table391 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray204 kitchen)
     (at tray152 kitchen)
     (at_kitchen_bread bread141)
     (at_kitchen_content content495)
     (not_allergic_gluten child265)
     (waiting child265 table361)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child265)
    )
  )
)
