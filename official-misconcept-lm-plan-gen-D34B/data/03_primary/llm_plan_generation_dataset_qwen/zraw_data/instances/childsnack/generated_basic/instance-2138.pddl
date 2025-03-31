; child-snack task with 2 children and 0.53 gluten factor 
; constant factor of 1.9
; random seed: 272833

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child335 child273 - child
    bread14 bread243 - bread-portion
    content402 content352 - content-portion
    tray483 tray254 tray11 - tray
    table443 table446 table362 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray483 kitchen)
     (at tray254 kitchen)
     (at tray11 kitchen)
     (at_kitchen_bread bread14)
     (at_kitchen_bread bread243)
     (at_kitchen_content content402)
     (at_kitchen_content content352)
     (no_gluten_bread bread243)
     (no_gluten_content content402)
     (allergic_gluten child273)
     (not_allergic_gluten child335)
     (waiting child335 table446)
     (waiting child273 table443)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child335)
     (served child273)
    )
  )
)
