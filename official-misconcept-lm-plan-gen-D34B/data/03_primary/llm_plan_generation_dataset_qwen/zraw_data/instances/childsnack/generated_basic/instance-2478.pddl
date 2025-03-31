; child-snack task with 2 children and 0.39 gluten factor 
; constant factor of 1.9
; random seed: 968338

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child484 child271 - child
    bread338 bread64 - bread-portion
    content315 content381 - content-portion
    tray270 tray274 - tray
    table203 table145 table13 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray270 kitchen)
     (at tray274 kitchen)
     (at_kitchen_bread bread338)
     (at_kitchen_bread bread64)
     (at_kitchen_content content315)
     (at_kitchen_content content381)
     (not_allergic_gluten child484)
     (not_allergic_gluten child271)
     (waiting child484 table13)
     (waiting child271 table13)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child484)
     (served child271)
    )
  )
)
