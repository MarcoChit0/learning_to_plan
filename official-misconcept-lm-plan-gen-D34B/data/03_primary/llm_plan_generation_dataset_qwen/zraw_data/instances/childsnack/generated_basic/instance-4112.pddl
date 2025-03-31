; child-snack task with 2 children and 0.44 gluten factor 
; constant factor of 1.9
; random seed: 770482

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child144 child8 - child
    bread32 bread178 - bread-portion
    content142 content154 - content-portion
    tray454 tray36 tray336 - tray
    table455 table234 table386 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray454 kitchen)
     (at tray36 kitchen)
     (at tray336 kitchen)
     (at_kitchen_bread bread32)
     (at_kitchen_bread bread178)
     (at_kitchen_content content142)
     (at_kitchen_content content154)
     (not_allergic_gluten child144)
     (not_allergic_gluten child8)
     (waiting child144 table455)
     (waiting child8 table455)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child144)
     (served child8)
    )
  )
)
