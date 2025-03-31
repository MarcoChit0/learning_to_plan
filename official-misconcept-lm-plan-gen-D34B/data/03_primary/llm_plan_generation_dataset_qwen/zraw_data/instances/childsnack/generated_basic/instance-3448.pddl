; child-snack task with 1 children and 0.77 gluten factor 
; constant factor of 1.9
; random seed: 818222

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child333 - child
    bread69 - bread-portion
    content69 - content-portion
    tray169 tray299 tray346 tray11 - tray
    table235 table220 table7 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray169 kitchen)
     (at tray299 kitchen)
     (at tray346 kitchen)
     (at tray11 kitchen)
     (at_kitchen_bread bread69)
     (at_kitchen_content content69)
     (not_allergic_gluten child333)
     (waiting child333 table7)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child333)
    )
  )
)
