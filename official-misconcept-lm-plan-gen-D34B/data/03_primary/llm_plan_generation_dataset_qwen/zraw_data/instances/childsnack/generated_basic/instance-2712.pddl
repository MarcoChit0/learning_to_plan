; child-snack task with 3 children and 0.23 gluten factor 
; constant factor of 1.9
; random seed: 56652

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child483 child380 child188 - child
    bread136 bread345 bread85 - bread-portion
    content299 content475 content429 - content-portion
    tray90 tray145 tray237 - tray
    table446 table337 table54 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 sandw6 - sandwich
  )
  (:init
     (at tray90 kitchen)
     (at tray145 kitchen)
     (at tray237 kitchen)
     (at_kitchen_bread bread136)
     (at_kitchen_bread bread345)
     (at_kitchen_bread bread85)
     (at_kitchen_content content299)
     (at_kitchen_content content475)
     (at_kitchen_content content429)
     (not_allergic_gluten child483)
     (not_allergic_gluten child188)
     (not_allergic_gluten child380)
     (waiting child483 table446)
     (waiting child380 table337)
     (waiting child188 table54)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
     (notexist sandw6)
  )
  (:goal
    (and
     (served child483)
     (served child380)
     (served child188)
    )
  )
)
