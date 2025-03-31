; child-snack task with 1 children and 0.62 gluten factor 
; constant factor of 1.5
; random seed: 100521

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child189 - child
    bread132 - bread-portion
    content195 - content-portion
    tray436 tray374 tray328 tray202 - tray
    table345 table411 table169 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray436 kitchen)
     (at tray374 kitchen)
     (at tray328 kitchen)
     (at tray202 kitchen)
     (at_kitchen_bread bread132)
     (at_kitchen_content content195)
     (not_allergic_gluten child189)
     (waiting child189 table345)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child189)
    )
  )
)
