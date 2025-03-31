; child-snack task with 2 children and 0.64 gluten factor 
; constant factor of 1.9
; random seed: 309162

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child190 child264 - child
    bread354 bread110 - bread-portion
    content69 content385 - content-portion
    tray492 tray142 - tray
    table216 table403 table229 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray492 kitchen)
     (at tray142 kitchen)
     (at_kitchen_bread bread354)
     (at_kitchen_bread bread110)
     (at_kitchen_content content69)
     (at_kitchen_content content385)
     (no_gluten_bread bread110)
     (no_gluten_content content69)
     (allergic_gluten child190)
     (not_allergic_gluten child264)
     (waiting child190 table216)
     (waiting child264 table229)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child190)
     (served child264)
    )
  )
)
