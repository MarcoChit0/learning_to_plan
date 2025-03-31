; child-snack task with 1 children and 0.73 gluten factor 
; constant factor of 1.5
; random seed: 440259

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child121 - child
    bread418 - bread-portion
    content101 - content-portion
    tray384 - tray
    table461 table171 table473 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray384 kitchen)
     (at_kitchen_bread bread418)
     (at_kitchen_content content101)
     (not_allergic_gluten child121)
     (waiting child121 table461)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child121)
    )
  )
)
