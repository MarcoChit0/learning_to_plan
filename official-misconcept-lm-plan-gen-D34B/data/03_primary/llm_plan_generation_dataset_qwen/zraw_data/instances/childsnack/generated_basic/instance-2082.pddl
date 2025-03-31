; child-snack task with 1 children and 0.62 gluten factor 
; constant factor of 1.5
; random seed: 834271

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child427 - child
    bread342 - bread-portion
    content416 - content-portion
    tray52 tray292 tray351 tray458 - tray
    table100 table148 table140 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray52 kitchen)
     (at tray292 kitchen)
     (at tray351 kitchen)
     (at tray458 kitchen)
     (at_kitchen_bread bread342)
     (at_kitchen_content content416)
     (not_allergic_gluten child427)
     (waiting child427 table148)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child427)
    )
  )
)
