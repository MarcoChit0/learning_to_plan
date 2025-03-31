; child-snack task with 2 children and 0.44 gluten factor 
; constant factor of 1.9
; random seed: 687691

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child256 child480 - child
    bread259 bread210 - bread-portion
    content92 content107 - content-portion
    tray114 tray260 tray56 - tray
    table387 table301 table453 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray114 kitchen)
     (at tray260 kitchen)
     (at tray56 kitchen)
     (at_kitchen_bread bread259)
     (at_kitchen_bread bread210)
     (at_kitchen_content content92)
     (at_kitchen_content content107)
     (not_allergic_gluten child480)
     (not_allergic_gluten child256)
     (waiting child256 table453)
     (waiting child480 table301)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child256)
     (served child480)
    )
  )
)
