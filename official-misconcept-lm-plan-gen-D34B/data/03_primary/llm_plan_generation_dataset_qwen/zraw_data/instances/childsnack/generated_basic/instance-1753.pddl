; child-snack task with 1 children and 0.36 gluten factor 
; constant factor of 1.3
; random seed: 608867

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child37 - child
    bread243 - bread-portion
    content185 - content-portion
    tray156 tray357 tray497 - tray
    table180 table268 table136 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray156 kitchen)
     (at tray357 kitchen)
     (at tray497 kitchen)
     (at_kitchen_bread bread243)
     (at_kitchen_content content185)
     (not_allergic_gluten child37)
     (waiting child37 table268)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child37)
    )
  )
)
