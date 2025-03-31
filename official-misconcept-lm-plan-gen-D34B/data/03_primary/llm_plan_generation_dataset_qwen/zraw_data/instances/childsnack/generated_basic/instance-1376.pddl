; child-snack task with 2 children and 0.27 gluten factor 
; constant factor of 1.7
; random seed: 465325

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child412 child311 - child
    bread119 bread73 - bread-portion
    content32 content347 - content-portion
    tray386 tray150 tray423 - tray
    table92 table411 table28 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray386 kitchen)
     (at tray150 kitchen)
     (at tray423 kitchen)
     (at_kitchen_bread bread119)
     (at_kitchen_bread bread73)
     (at_kitchen_content content32)
     (at_kitchen_content content347)
     (not_allergic_gluten child412)
     (not_allergic_gluten child311)
     (waiting child412 table411)
     (waiting child311 table28)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child412)
     (served child311)
    )
  )
)
