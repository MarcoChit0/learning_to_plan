; child-snack task with 3 children and 0.33 gluten factor 
; constant factor of 1.7
; random seed: 715109

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child9 child231 child465 - child
    bread88 bread446 bread256 - bread-portion
    content96 content23 content410 - content-portion
    tray215 tray488 tray366 tray184 - tray
    table3 table279 table471 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 sandw6 - sandwich
  )
  (:init
     (at tray215 kitchen)
     (at tray488 kitchen)
     (at tray366 kitchen)
     (at tray184 kitchen)
     (at_kitchen_bread bread88)
     (at_kitchen_bread bread446)
     (at_kitchen_bread bread256)
     (at_kitchen_content content96)
     (at_kitchen_content content23)
     (at_kitchen_content content410)
     (not_allergic_gluten child465)
     (not_allergic_gluten child9)
     (not_allergic_gluten child231)
     (waiting child9 table3)
     (waiting child231 table279)
     (waiting child465 table279)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
     (notexist sandw6)
  )
  (:goal
    (and
     (served child9)
     (served child231)
     (served child465)
    )
  )
)
