; child-snack task with 1 children and 0.43 gluten factor 
; constant factor of 1.9
; random seed: 543570

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child120 - child
    bread399 - bread-portion
    content349 - content-portion
    tray40 tray274 tray294 tray222 - tray
    table394 table37 table259 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray40 kitchen)
     (at tray274 kitchen)
     (at tray294 kitchen)
     (at tray222 kitchen)
     (at_kitchen_bread bread399)
     (at_kitchen_content content349)
     (not_allergic_gluten child120)
     (waiting child120 table394)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child120)
    )
  )
)
