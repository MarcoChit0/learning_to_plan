; child-snack task with 2 children and 0.28 gluten factor 
; constant factor of 1.7
; random seed: 38788

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child324 child431 - child
    bread362 bread337 - bread-portion
    content342 content119 - content-portion
    tray259 - tray
    table273 table398 table7 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray259 kitchen)
     (at_kitchen_bread bread362)
     (at_kitchen_bread bread337)
     (at_kitchen_content content342)
     (at_kitchen_content content119)
     (not_allergic_gluten child324)
     (not_allergic_gluten child431)
     (waiting child324 table273)
     (waiting child431 table273)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child324)
     (served child431)
    )
  )
)
