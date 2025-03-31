; child-snack task with 2 children and 0.79 gluten factor 
; constant factor of 1.9
; random seed: 251326

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child265 child301 - child
    bread150 bread267 - bread-portion
    content101 content373 - content-portion
    tray321 tray160 tray64 tray335 - tray
    table400 table265 table450 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray321 kitchen)
     (at tray160 kitchen)
     (at tray64 kitchen)
     (at tray335 kitchen)
     (at_kitchen_bread bread150)
     (at_kitchen_bread bread267)
     (at_kitchen_content content101)
     (at_kitchen_content content373)
     (no_gluten_bread bread267)
     (no_gluten_content content373)
     (allergic_gluten child265)
     (not_allergic_gluten child301)
     (waiting child265 table265)
     (waiting child301 table450)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child265)
     (served child301)
    )
  )
)
