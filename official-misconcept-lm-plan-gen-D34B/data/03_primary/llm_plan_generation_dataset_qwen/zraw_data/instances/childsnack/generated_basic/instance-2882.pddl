; child-snack task with 1 children and 0.23 gluten factor 
; constant factor of 1.7
; random seed: 92314

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child124 - child
    bread241 - bread-portion
    content452 - content-portion
    tray248 tray474 - tray
    table226 table461 table110 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray248 kitchen)
     (at tray474 kitchen)
     (at_kitchen_bread bread241)
     (at_kitchen_content content452)
     (not_allergic_gluten child124)
     (waiting child124 table461)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child124)
    )
  )
)
