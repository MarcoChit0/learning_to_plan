; child-snack task with 2 children and 0.44 gluten factor 
; constant factor of 1.9
; random seed: 308262

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child53 child425 - child
    bread368 bread321 - bread-portion
    content424 content475 - content-portion
    tray397 tray161 tray210 - tray
    table200 table93 table299 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray397 kitchen)
     (at tray161 kitchen)
     (at tray210 kitchen)
     (at_kitchen_bread bread368)
     (at_kitchen_bread bread321)
     (at_kitchen_content content424)
     (at_kitchen_content content475)
     (not_allergic_gluten child425)
     (not_allergic_gluten child53)
     (waiting child53 table299)
     (waiting child425 table93)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child53)
     (served child425)
    )
  )
)
