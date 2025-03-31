; child-snack task with 2 children and 0.28 gluten factor 
; constant factor of 1.7
; random seed: 256351

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child457 child364 - child
    bread473 bread13 - bread-portion
    content299 content198 - content-portion
    tray477 - tray
    table452 table90 table264 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray477 kitchen)
     (at_kitchen_bread bread473)
     (at_kitchen_bread bread13)
     (at_kitchen_content content299)
     (at_kitchen_content content198)
     (not_allergic_gluten child364)
     (not_allergic_gluten child457)
     (waiting child457 table264)
     (waiting child364 table90)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child457)
     (served child364)
    )
  )
)
