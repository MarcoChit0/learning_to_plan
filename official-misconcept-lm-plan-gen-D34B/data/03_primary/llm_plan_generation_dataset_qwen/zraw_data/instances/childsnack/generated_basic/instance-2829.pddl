; child-snack task with 2 children and 0.32 gluten factor 
; constant factor of 1.7
; random seed: 199666

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child253 child235 - child
    bread96 bread279 - bread-portion
    content238 content345 - content-portion
    tray133 tray370 - tray
    table321 table61 table191 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray133 kitchen)
     (at tray370 kitchen)
     (at_kitchen_bread bread96)
     (at_kitchen_bread bread279)
     (at_kitchen_content content238)
     (at_kitchen_content content345)
     (not_allergic_gluten child253)
     (not_allergic_gluten child235)
     (waiting child253 table321)
     (waiting child235 table61)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child253)
     (served child235)
    )
  )
)
