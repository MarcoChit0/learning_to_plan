; child-snack task with 1 children and 0.43 gluten factor 
; constant factor of 1.9
; random seed: 762308

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child30 - child
    bread364 - bread-portion
    content334 - content-portion
    tray346 tray254 tray227 tray414 - tray
    table28 table159 table207 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray346 kitchen)
     (at tray254 kitchen)
     (at tray227 kitchen)
     (at tray414 kitchen)
     (at_kitchen_bread bread364)
     (at_kitchen_content content334)
     (not_allergic_gluten child30)
     (waiting child30 table28)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child30)
    )
  )
)
