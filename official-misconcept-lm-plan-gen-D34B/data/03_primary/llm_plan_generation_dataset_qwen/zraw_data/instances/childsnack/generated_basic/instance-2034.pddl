; child-snack task with 2 children and 0.34 gluten factor 
; constant factor of 1.5
; random seed: 515352

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child113 child468 - child
    bread280 bread430 - bread-portion
    content476 content221 - content-portion
    tray299 tray36 tray386 - tray
    table146 table428 table124 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray299 kitchen)
     (at tray36 kitchen)
     (at tray386 kitchen)
     (at_kitchen_bread bread280)
     (at_kitchen_bread bread430)
     (at_kitchen_content content476)
     (at_kitchen_content content221)
     (not_allergic_gluten child468)
     (not_allergic_gluten child113)
     (waiting child113 table124)
     (waiting child468 table146)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child113)
     (served child468)
    )
  )
)
