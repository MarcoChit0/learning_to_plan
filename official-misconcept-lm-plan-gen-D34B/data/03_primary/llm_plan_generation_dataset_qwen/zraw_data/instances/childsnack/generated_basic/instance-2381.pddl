; child-snack task with 2 children and 0.38 gluten factor 
; constant factor of 1.9
; random seed: 164253

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child314 child424 - child
    bread231 bread444 - bread-portion
    content372 content435 - content-portion
    tray282 - tray
    table483 table18 table217 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray282 kitchen)
     (at_kitchen_bread bread231)
     (at_kitchen_bread bread444)
     (at_kitchen_content content372)
     (at_kitchen_content content435)
     (not_allergic_gluten child314)
     (not_allergic_gluten child424)
     (waiting child314 table483)
     (waiting child424 table18)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child314)
     (served child424)
    )
  )
)
