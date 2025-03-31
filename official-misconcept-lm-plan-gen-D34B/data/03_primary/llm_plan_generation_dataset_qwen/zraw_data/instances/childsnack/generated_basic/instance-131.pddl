; child-snack task with 2 children and 0.33 gluten factor 
; constant factor of 1.7
; random seed: 602043

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child484 child215 - child
    bread199 bread58 - bread-portion
    content262 content6 - content-portion
    tray393 tray384 tray91 tray100 - tray
    table284 table470 table271 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray393 kitchen)
     (at tray384 kitchen)
     (at tray91 kitchen)
     (at tray100 kitchen)
     (at_kitchen_bread bread199)
     (at_kitchen_bread bread58)
     (at_kitchen_content content262)
     (at_kitchen_content content6)
     (not_allergic_gluten child215)
     (not_allergic_gluten child484)
     (waiting child484 table284)
     (waiting child215 table284)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child484)
     (served child215)
    )
  )
)
