; child-snack task with 1 children and 0.75 gluten factor 
; constant factor of 1.7
; random seed: 902789

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child124 - child
    bread292 - bread-portion
    content419 - content-portion
    tray364 - tray
    table196 table162 table198 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray364 kitchen)
     (at_kitchen_bread bread292)
     (at_kitchen_content content419)
     (not_allergic_gluten child124)
     (waiting child124 table198)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child124)
    )
  )
)
