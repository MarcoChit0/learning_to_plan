; child-snack task with 1 children and 0.43 gluten factor 
; constant factor of 1.9
; random seed: 710015

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child215 - child
    bread97 - bread-portion
    content125 - content-portion
    tray17 tray387 tray496 tray368 - tray
    table152 table473 table325 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray17 kitchen)
     (at tray387 kitchen)
     (at tray496 kitchen)
     (at tray368 kitchen)
     (at_kitchen_bread bread97)
     (at_kitchen_content content125)
     (not_allergic_gluten child215)
     (waiting child215 table473)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child215)
    )
  )
)
