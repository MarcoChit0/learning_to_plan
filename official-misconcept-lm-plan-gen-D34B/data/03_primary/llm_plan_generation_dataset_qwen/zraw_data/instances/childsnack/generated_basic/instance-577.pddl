; child-snack task with 1 children and 0.63 gluten factor 
; constant factor of 1.5
; random seed: 204490

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child256 - child
    bread224 - bread-portion
    content378 - content-portion
    tray333 - tray
    table144 table453 table38 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray333 kitchen)
     (at_kitchen_bread bread224)
     (at_kitchen_content content378)
     (not_allergic_gluten child256)
     (waiting child256 table453)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child256)
    )
  )
)
