; child-snack task with 2 children and 0.32 gluten factor 
; constant factor of 1.7
; random seed: 492017

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child495 child40 - child
    bread296 bread52 - bread-portion
    content278 content413 - content-portion
    tray400 tray116 - tray
    table290 table476 table145 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray400 kitchen)
     (at tray116 kitchen)
     (at_kitchen_bread bread296)
     (at_kitchen_bread bread52)
     (at_kitchen_content content278)
     (at_kitchen_content content413)
     (not_allergic_gluten child40)
     (not_allergic_gluten child495)
     (waiting child495 table476)
     (waiting child40 table290)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child495)
     (served child40)
    )
  )
)
