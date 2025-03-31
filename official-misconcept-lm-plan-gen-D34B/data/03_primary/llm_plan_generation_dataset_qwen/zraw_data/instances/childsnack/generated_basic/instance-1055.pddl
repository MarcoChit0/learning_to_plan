; child-snack task with 2 children and 0.41 gluten factor 
; constant factor of 1.3
; random seed: 306015

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child42 child31 - child
    bread340 bread442 - bread-portion
    content346 content437 - content-portion
    tray49 tray465 tray64 - tray
    table76 table223 table354 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray49 kitchen)
     (at tray465 kitchen)
     (at tray64 kitchen)
     (at_kitchen_bread bread340)
     (at_kitchen_bread bread442)
     (at_kitchen_content content346)
     (at_kitchen_content content437)
     (not_allergic_gluten child31)
     (not_allergic_gluten child42)
     (waiting child42 table76)
     (waiting child31 table223)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child42)
     (served child31)
    )
  )
)
