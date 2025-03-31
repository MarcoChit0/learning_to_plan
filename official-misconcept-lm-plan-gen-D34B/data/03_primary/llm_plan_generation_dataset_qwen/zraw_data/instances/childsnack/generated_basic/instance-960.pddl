; child-snack task with 3 children and 0.33 gluten factor 
; constant factor of 1.7
; random seed: 366403

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child460 child335 child468 - child
    bread243 bread20 bread221 - bread-portion
    content333 content120 content195 - content-portion
    tray234 tray293 tray43 tray493 - tray
    table154 table312 table180 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 sandw6 - sandwich
  )
  (:init
     (at tray234 kitchen)
     (at tray293 kitchen)
     (at tray43 kitchen)
     (at tray493 kitchen)
     (at_kitchen_bread bread243)
     (at_kitchen_bread bread20)
     (at_kitchen_bread bread221)
     (at_kitchen_content content333)
     (at_kitchen_content content120)
     (at_kitchen_content content195)
     (not_allergic_gluten child460)
     (not_allergic_gluten child335)
     (not_allergic_gluten child468)
     (waiting child460 table154)
     (waiting child335 table312)
     (waiting child468 table180)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
     (notexist sandw6)
  )
  (:goal
    (and
     (served child460)
     (served child335)
     (served child468)
    )
  )
)
