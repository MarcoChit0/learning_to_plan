; child-snack task with 1 children and 0.4 gluten factor 
; constant factor of 1.5
; random seed: 212376

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child156 - child
    bread112 - bread-portion
    content225 - content-portion
    tray358 tray334 - tray
    table176 table322 table254 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray358 kitchen)
     (at tray334 kitchen)
     (at_kitchen_bread bread112)
     (at_kitchen_content content225)
     (not_allergic_gluten child156)
     (waiting child156 table254)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child156)
    )
  )
)
