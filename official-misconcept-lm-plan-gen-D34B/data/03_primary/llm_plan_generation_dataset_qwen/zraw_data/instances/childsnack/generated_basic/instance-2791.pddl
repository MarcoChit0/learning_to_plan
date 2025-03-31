; child-snack task with 2 children and 0.32 gluten factor 
; constant factor of 1.7
; random seed: 756321

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child124 child419 - child
    bread468 bread493 - bread-portion
    content354 content301 - content-portion
    tray173 tray411 - tray
    table19 table331 table495 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray173 kitchen)
     (at tray411 kitchen)
     (at_kitchen_bread bread468)
     (at_kitchen_bread bread493)
     (at_kitchen_content content354)
     (at_kitchen_content content301)
     (not_allergic_gluten child124)
     (not_allergic_gluten child419)
     (waiting child124 table495)
     (waiting child419 table331)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child124)
     (served child419)
    )
  )
)
