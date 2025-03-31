; child-snack task with 1 children and 0.3 gluten factor 
; constant factor of 1.3
; random seed: 478523

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child198 - child
    bread28 - bread-portion
    content145 - content-portion
    tray3 tray255 tray36 - tray
    table213 table351 table145 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray3 kitchen)
     (at tray255 kitchen)
     (at tray36 kitchen)
     (at_kitchen_bread bread28)
     (at_kitchen_content content145)
     (not_allergic_gluten child198)
     (waiting child198 table145)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child198)
    )
  )
)
