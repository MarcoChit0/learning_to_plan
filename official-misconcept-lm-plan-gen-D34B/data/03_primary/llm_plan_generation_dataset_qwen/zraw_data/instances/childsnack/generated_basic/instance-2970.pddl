; child-snack task with 1 children and 0.69 gluten factor 
; constant factor of 1.3
; random seed: 48903

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child327 - child
    bread113 - bread-portion
    content136 - content-portion
    tray444 tray160 tray155 tray305 - tray
    table168 table69 table180 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray444 kitchen)
     (at tray160 kitchen)
     (at tray155 kitchen)
     (at tray305 kitchen)
     (at_kitchen_bread bread113)
     (at_kitchen_content content136)
     (not_allergic_gluten child327)
     (waiting child327 table168)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child327)
    )
  )
)
