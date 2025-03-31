; child-snack task with 1 children and 0.7 gluten factor 
; constant factor of 1.3
; random seed: 326712

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child301 - child
    bread216 - bread-portion
    content168 - content-portion
    tray376 - tray
    table474 table378 table343 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray376 kitchen)
     (at_kitchen_bread bread216)
     (at_kitchen_content content168)
     (not_allergic_gluten child301)
     (waiting child301 table343)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child301)
    )
  )
)
