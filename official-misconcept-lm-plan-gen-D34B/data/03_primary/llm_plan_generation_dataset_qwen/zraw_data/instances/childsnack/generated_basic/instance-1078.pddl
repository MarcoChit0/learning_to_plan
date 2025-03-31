; child-snack task with 2 children and 0.41 gluten factor 
; constant factor of 1.3
; random seed: 321169

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child301 child408 - child
    bread448 bread207 - bread-portion
    content290 content338 - content-portion
    tray478 tray64 tray308 - tray
    table278 table213 table92 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray478 kitchen)
     (at tray64 kitchen)
     (at tray308 kitchen)
     (at_kitchen_bread bread448)
     (at_kitchen_bread bread207)
     (at_kitchen_content content290)
     (at_kitchen_content content338)
     (not_allergic_gluten child301)
     (not_allergic_gluten child408)
     (waiting child301 table92)
     (waiting child408 table213)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child301)
     (served child408)
    )
  )
)
