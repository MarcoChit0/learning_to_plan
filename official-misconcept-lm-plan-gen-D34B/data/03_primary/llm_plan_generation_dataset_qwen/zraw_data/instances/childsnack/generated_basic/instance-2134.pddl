; child-snack task with 2 children and 0.53 gluten factor 
; constant factor of 1.9
; random seed: 178546

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child311 child336 - child
    bread178 bread465 - bread-portion
    content101 content249 - content-portion
    tray28 tray58 tray381 - tray
    table336 table84 table323 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray28 kitchen)
     (at tray58 kitchen)
     (at tray381 kitchen)
     (at_kitchen_bread bread178)
     (at_kitchen_bread bread465)
     (at_kitchen_content content101)
     (at_kitchen_content content249)
     (no_gluten_bread bread465)
     (no_gluten_content content249)
     (allergic_gluten child311)
     (not_allergic_gluten child336)
     (waiting child311 table336)
     (waiting child336 table323)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child311)
     (served child336)
    )
  )
)
