; child-snack task with 3 children and 0.22 gluten factor 
; constant factor of 1.9
; random seed: 148600

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child259 child383 child429 - child
    bread456 bread70 bread10 - bread-portion
    content269 content63 content481 - content-portion
    tray239 - tray
    table277 table23 table322 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 sandw6 - sandwich
  )
  (:init
     (at tray239 kitchen)
     (at_kitchen_bread bread456)
     (at_kitchen_bread bread70)
     (at_kitchen_bread bread10)
     (at_kitchen_content content269)
     (at_kitchen_content content63)
     (at_kitchen_content content481)
     (not_allergic_gluten child259)
     (not_allergic_gluten child429)
     (not_allergic_gluten child383)
     (waiting child259 table23)
     (waiting child383 table277)
     (waiting child429 table23)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
     (notexist sandw6)
  )
  (:goal
    (and
     (served child259)
     (served child383)
     (served child429)
    )
  )
)
