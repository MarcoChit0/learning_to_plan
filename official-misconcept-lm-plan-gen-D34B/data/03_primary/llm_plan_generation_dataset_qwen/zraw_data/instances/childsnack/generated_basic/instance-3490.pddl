; child-snack task with 2 children and 0.63 gluten factor 
; constant factor of 1.7
; random seed: 657478

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child31 child299 - child
    bread465 bread191 - bread-portion
    content465 content435 - content-portion
    tray15 tray112 - tray
    table58 table112 table163 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray15 kitchen)
     (at tray112 kitchen)
     (at_kitchen_bread bread465)
     (at_kitchen_bread bread191)
     (at_kitchen_content content465)
     (at_kitchen_content content435)
     (no_gluten_bread bread191)
     (no_gluten_content content465)
     (allergic_gluten child299)
     (not_allergic_gluten child31)
     (waiting child31 table163)
     (waiting child299 table163)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child31)
     (served child299)
    )
  )
)
