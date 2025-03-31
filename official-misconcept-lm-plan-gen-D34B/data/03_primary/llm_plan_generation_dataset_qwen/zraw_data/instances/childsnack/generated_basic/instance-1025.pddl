; child-snack task with 3 children and 0.3 gluten factor 
; constant factor of 1.3
; random seed: 918206

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child183 child77 child253 - child
    bread58 bread246 bread60 - bread-portion
    content189 content313 content288 - content-portion
    tray287 - tray
    table57 table480 table395 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray287 kitchen)
     (at_kitchen_bread bread58)
     (at_kitchen_bread bread246)
     (at_kitchen_bread bread60)
     (at_kitchen_content content189)
     (at_kitchen_content content313)
     (at_kitchen_content content288)
     (not_allergic_gluten child183)
     (not_allergic_gluten child253)
     (not_allergic_gluten child77)
     (waiting child183 table480)
     (waiting child77 table395)
     (waiting child253 table480)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child183)
     (served child77)
     (served child253)
    )
  )
)
